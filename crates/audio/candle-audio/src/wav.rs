//! 16-bit PCM WAV encoding of a [`gen_core::AudioTrack`] (sc-12835) — the audio lane's
//! sibling of the media families' image/video encode step. A provider synthesizes float
//! samples into [`gen_core::GenerationOutput::Audio`]; a consumer (worker, tests) persists
//! them here. Deliberately minimal: canonical 44-byte-header RIFF/WAVE, PCM 16-bit,
//! interleaved — the one container every audio toolchain reads. No decode path (the
//! reference-audio *input* path resamples through provider-owned preprocessing).

use std::path::Path;

use gen_core::AudioTrack;

use crate::{AudioError, Result};

/// Encode a track as a complete 16-bit PCM WAV byte stream. Samples are clamped to
/// `[-1, 1]` then scaled to `i16` (the standard float→PCM convention); `track.samples`
/// is channel-interleaved, matching the [`AudioTrack`] contract.
pub fn encode_wav_pcm16(track: &AudioTrack) -> Result<Vec<u8>> {
    if track.channels == 0 || track.sample_rate == 0 {
        return Err(AudioError::Msg(format!(
            "WAV encode needs channels >= 1 and sample_rate >= 1 (got {} ch @ {} Hz)",
            track.channels, track.sample_rate
        )));
    }
    if !track.samples.len().is_multiple_of(track.channels as usize) {
        return Err(AudioError::Msg(format!(
            "WAV encode: {} samples do not divide into {} interleaved channels",
            track.samples.len(),
            track.channels
        )));
    }
    // The RIFF chunk size field stores `36 + data_len`, so data alone must leave headroom for
    // the 36 header bytes — a bare `<= u32::MAX` bound would overflow the `36 +` below for a
    // track within 36 bytes of 4 GiB.
    let data_len = track
        .samples
        .len()
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n <= u32::MAX - 36)
        .ok_or_else(|| {
            AudioError::Msg(format!(
                "WAV encode: {} samples exceed the 32-bit RIFF size field",
                track.samples.len()
            ))
        })?;
    let byte_rate = track.sample_rate * track.channels as u32 * 2;
    let block_align = track.channels * 2;

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&track.channels.to_le_bytes());
    out.extend_from_slice(&track.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in &track.samples {
        let clamped = s.clamp(-1.0, 1.0);
        out.extend_from_slice(&((clamped * i16::MAX as f32) as i16).to_le_bytes());
    }
    Ok(out)
}

/// [`encode_wav_pcm16`] straight to a file.
pub fn write_wav_pcm16(path: &Path, track: &AudioTrack) -> Result<()> {
    let bytes = encode_wav_pcm16(track)?;
    std::fs::write(path, bytes)
        .map_err(|e| AudioError::Msg(format!("write WAV {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(samples: Vec<f32>, sample_rate: u32, channels: u16) -> AudioTrack {
        AudioTrack {
            samples,
            sample_rate,
            channels,
            ..Default::default()
        }
    }

    #[test]
    fn encodes_the_canonical_pcm16_header() {
        let bytes = encode_wav_pcm16(&track(vec![0.0, 1.0, -1.0, 0.5], 24_000, 1)).unwrap();
        assert_eq!(bytes.len(), 44 + 8);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1); // PCM
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1); // mono
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            24_000
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            48_000 // byte rate = sr * ch * 2
        );
        assert_eq!(u16::from_le_bytes(bytes[32..34].try_into().unwrap()), 2); // block align
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16); // bits
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 8);
    }

    #[test]
    fn scales_and_clamps_samples() {
        let bytes = encode_wav_pcm16(&track(vec![0.0, 1.0, -1.0, 2.0], 8_000, 1)).unwrap();
        let pcm: Vec<i16> = bytes[44..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], i16::MAX);
        assert_eq!(pcm[2], -i16::MAX); // symmetric scale, clamped at -1.0
        assert_eq!(pcm[3], i16::MAX); // out-of-range input clamps, never wraps
    }

    #[test]
    fn stereo_interleaving_is_preserved_and_writes_to_disk() {
        let t = track(vec![0.25, -0.25, 0.5, -0.5], 44_100, 2);
        let bytes = encode_wav_pcm16(&t).unwrap();
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
        let dir = std::env::temp_dir().join("candle-audio-wav-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo.wav");
        write_wav_pcm16(&path, &t).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_malformed_tracks() {
        assert!(encode_wav_pcm16(&track(vec![0.0], 24_000, 0)).is_err()); // zero channels
        assert!(encode_wav_pcm16(&track(vec![0.0], 0, 1)).is_err()); // zero rate
        assert!(encode_wav_pcm16(&track(vec![0.0; 3], 24_000, 2)).is_err()); // ragged interleave
    }
}
