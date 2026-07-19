//! The streaming PCM-chunking mechanism (sc-13334).
//!
//! MOSS-TTS-Realtime is autoregressive: the AR brain emits one RVQ frame at a time, and the codec
//! (once ported) decodes each block of frames into a block of PCM. The provider's
//! `gen_core::Generator::generate_streaming` emits one `gen_core::AudioChunk` per decoded PCM
//! block, so a consumer starts playback well before the full track finishes. This module holds the
//! block→chunk mechanism as a small pure helper so it is unit-testable **offline** (no weights, no
//! codec): given the final PCM and a block size it produces contiguous, frame-aligned,
//! `0..N`-indexed chunks that reassemble byte-for-byte to the track — exactly the invariant the
//! shared `check_audio_streaming` conformance enforces. When the codec lands, the streaming path
//! emits these blocks incrementally as they are decoded rather than after the fact.

use candle_audio::gen_core::{AudioChunk, AudioTrack};

/// Split a finished [`AudioTrack`] into `0..N`-indexed [`AudioChunk`]s of at most
/// `frames_per_chunk` audio frames each (a *frame* is one sample per channel). The chunks are
/// contiguous, each carries the track's `sample_rate`/`channels`, each is a whole number of frames,
/// and concatenating their `samples` in index order reproduces the track's `samples` exactly (the
/// reassembly law). `frames_per_chunk` is clamped to at least 1.
pub fn into_chunks(track: &AudioTrack, frames_per_chunk: usize) -> Vec<AudioChunk> {
    let channels = track.channels.max(1) as usize;
    let frames_per_chunk = frames_per_chunk.max(1);
    let samples_per_chunk = frames_per_chunk * channels;
    if track.samples.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut index = 0;
    let mut offset = 0;
    while offset < track.samples.len() {
        let end = (offset + samples_per_chunk).min(track.samples.len());
        chunks.push(AudioChunk {
            samples: track.samples[offset..end].to_vec(),
            sample_rate: track.sample_rate,
            channels: track.channels,
            index,
        });
        index += 1;
        offset = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(samples: Vec<f32>, channels: u16) -> AudioTrack {
        AudioTrack {
            samples,
            sample_rate: 24_000,
            channels,
            ..Default::default()
        }
    }

    #[test]
    fn chunks_reassemble_and_are_indexed_and_frame_aligned() {
        let t = track((0..20).map(|i| i as f32).collect(), 1);
        let chunks = into_chunks(&t, 7);
        // 20 samples / 7 per chunk = 3 chunks (7, 7, 6).
        assert_eq!(chunks.len(), 3);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i, "contiguous 0..N indices");
            assert_eq!(c.sample_rate, t.sample_rate);
            assert_eq!(c.channels, t.channels);
            assert!(!c.samples.is_empty());
        }
        let reassembled: Vec<f32> = chunks.iter().flat_map(|c| c.samples.clone()).collect();
        assert_eq!(reassembled, t.samples, "reassembly law");
    }

    #[test]
    fn stereo_chunks_are_whole_frames() {
        // 6 frames of stereo = 12 interleaved samples; 2 frames per chunk → 3 chunks of 4 samples.
        let t = track((0..12).map(|i| i as f32).collect(), 2);
        let chunks = into_chunks(&t, 2);
        assert_eq!(chunks.len(), 3);
        for c in &chunks {
            assert_eq!(c.samples.len() % 2, 0, "whole stereo frames");
        }
        let reassembled: Vec<f32> = chunks.iter().flat_map(|c| c.samples.clone()).collect();
        assert_eq!(reassembled, t.samples);
    }

    #[test]
    fn multiple_chunks_before_completion_and_none_holds_whole_track() {
        // The incrementality invariants the streaming conformance enforces for a streaming provider.
        let t = track((0..100).map(|i| i as f32).collect(), 1);
        let chunks = into_chunks(&t, 16);
        assert!(chunks.len() >= 2, "at least two chunks");
        assert!(
            chunks.iter().all(|c| c.samples.len() < t.samples.len()),
            "no single chunk carries the whole track"
        );
    }

    #[test]
    fn empty_track_yields_no_chunks() {
        let t = track(vec![], 1);
        assert!(into_chunks(&t, 8).is_empty());
    }
}
