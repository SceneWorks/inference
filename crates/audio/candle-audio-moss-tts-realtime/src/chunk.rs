//! Interleaved streaming: RVQ frames → PCM chunks, decoded block-wise **as the AR loop runs**
//! (sc-13392).
//!
//! MOSS-TTS-Realtime is autoregressive: [`crate::decode`]'s AR loop emits one RVQ frame at a time.
//! Rather than wait for the whole track and chunk it after the fact, [`StreamingChunker`] decodes
//! the codec ([`crate::codec`]) over the growing frame prefix on every `block`-frame boundary and
//! emits the newly-revealed PCM as the next [`AudioChunk`] — so chunks flow **while** the AR loop is
//! still generating later frames. [`crate::model`] feeds it each frame from inside
//! [`crate::decode::Decoder::run`].
//!
//! The returned track is exactly the **concatenation of the emitted chunk tails** — the same
//! growing-prefix-delta numerical path the pre-interleaving code used, only now driven from inside
//! the AR loop. So the `AudioChunk` reassembly law (concat(chunks) == track) holds by construction,
//! and one-shot `generate()` is byte-identical to the concatenated stream because both entry points
//! build the track this way from the same prefixes+seed. (The MOSS codec is causal in principle but
//! not bit-exact across prefix lengths, so the track is assembled from the deltas rather than
//! replaced by a single full-length decode — that would perturb the samples at block seams.)
//!
//! The codec is abstracted behind [`PrefixDecoder`] so this interleaving logic is unit-testable
//! **offline** (no weights, no real codec) with a fake frame→PCM source.

use candle_audio::candle_core::Result as CandleResult;
use candle_audio::gen_core::{AudioChunk, AudioTrack};

use crate::decode::RvqFrame;

/// Decodes a growing prefix of RVQ frames into that prefix's full mono PCM. The production impl is
/// the MOSS-Audio-Tokenizer codec ([`crate::codec::MossAudioCodec`]); a fake drives the offline
/// [`StreamingChunker`] test.
pub trait PrefixDecoder {
    /// Native PCM sample rate (Hz).
    fn sample_rate(&self) -> u32;
    /// Waveform samples produced per RVQ frame (the codec's downsample rate).
    fn samples_per_frame(&self) -> usize;
    /// Decode `frames` → interleaved mono PCM of exactly `frames.len() * samples_per_frame()`
    /// samples, or `Ok(None)` if `cancel` tripped. Must be **causal**: `decode_prefix(&frames[..k])`
    /// is a byte-identical prefix of `decode_prefix(&frames[..k'])` for `k <= k'` — the property the
    /// reassembly law relies on.
    fn decode_prefix(
        &self,
        frames: &[RvqFrame],
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Vec<f32>>>;
}

/// Block-wise interleaved chunker (see the module docs). Feed it AR frames with [`push`](Self::push)
/// as they are produced; call [`finish`](Self::finish) at EOS/budget to flush the remainder and get
/// the full [`AudioTrack`]. It emits `0..N`-indexed, contiguous [`AudioChunk`]s whose concatenation
/// is exactly the track (the reassembly law).
pub struct StreamingChunker<'a, D: PrefixDecoder> {
    decoder: &'a D,
    block: usize,
    /// The RVQ frames buffered so far (the growing prefix handed to the codec).
    frames: Vec<RvqFrame>,
    /// The concatenation of every emitted chunk tail — the running track PCM (so the reassembly law
    /// holds by construction). Its length is also the number of samples already emitted.
    samples: Vec<f32>,
    /// Frame count covered by the last flush, so [`finish`](Self::finish) skips a redundant decode
    /// when the last `push` already landed on a block boundary.
    flushed_frames: usize,
    /// Next chunk index (`0..N`).
    index: usize,
}

impl<'a, D: PrefixDecoder> StreamingChunker<'a, D> {
    /// New chunker emitting a chunk every `block` frames (clamped to `>= 1`).
    pub fn new(decoder: &'a D, block: usize) -> Self {
        Self {
            decoder,
            block: block.max(1),
            frames: Vec::new(),
            samples: Vec::new(),
            flushed_frames: 0,
            index: 0,
        }
    }

    /// Buffer one AR frame; on a `block`-frame boundary decode the growing prefix and emit the newly
    /// revealed PCM as the next chunk. Returns `Ok(None)` if `cancel` tripped inside the codec decode.
    pub fn push(
        &mut self,
        frame: RvqFrame,
        cancel: &dyn Fn() -> bool,
        on_chunk: &mut dyn FnMut(AudioChunk),
    ) -> CandleResult<Option<()>> {
        self.frames.push(frame);
        if self.frames.len().is_multiple_of(self.block) && self.flush(cancel, on_chunk)?.is_none() {
            return Ok(None);
        }
        Ok(Some(()))
    }

    /// Flush any frames not yet covered by a chunk and return the full [`AudioTrack`]. `Ok(None)` on
    /// cancel.
    pub fn finish(
        mut self,
        cancel: &dyn Fn() -> bool,
        on_chunk: &mut dyn FnMut(AudioChunk),
    ) -> CandleResult<Option<AudioTrack>> {
        if self.flushed_frames != self.frames.len() && self.flush(cancel, on_chunk)?.is_none() {
            return Ok(None);
        }
        Ok(Some(AudioTrack {
            samples: self.samples,
            sample_rate: self.decoder.sample_rate(),
            channels: 1,
            stems: Vec::new(),
        }))
    }

    /// Decode the current frame prefix and emit the newly-revealed tail (the delta beyond what
    /// earlier chunks already carried) as one chunk, appending it to the running track. `Ok(None)`
    /// on cancel.
    fn flush(
        &mut self,
        cancel: &dyn Fn() -> bool,
        on_chunk: &mut dyn FnMut(AudioChunk),
    ) -> CandleResult<Option<()>> {
        let pcm = match self.decoder.decode_prefix(&self.frames, cancel)? {
            Some(p) => p,
            None => return Ok(None),
        };
        self.flushed_frames = self.frames.len();
        // The delta beyond what earlier prefixes already emitted (the growing-prefix numerical path).
        let tail = &pcm[self.samples.len().min(pcm.len())..];
        if !tail.is_empty() {
            on_chunk(AudioChunk {
                samples: tail.to_vec(),
                sample_rate: self.decoder.sample_rate(),
                channels: 1,
                index: self.index,
            });
            self.samples.extend_from_slice(tail);
            self.index += 1;
        }
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake, causal frame→PCM codec: frame `f`'s codebook-0 code `c` contributes `spf` samples all
    /// equal to `c as f32`, appended in frame order. `decode_prefix(&frames[..k])` is therefore a
    /// byte-identical prefix of any longer decode — exactly the causal contract the real codec meets.
    struct FakeCodec {
        spf: usize,
        cancel_after: Option<usize>,
    }

    impl PrefixDecoder for FakeCodec {
        fn sample_rate(&self) -> u32 {
            24_000
        }
        fn samples_per_frame(&self) -> usize {
            self.spf
        }
        fn decode_prefix(
            &self,
            frames: &[RvqFrame],
            cancel: &dyn Fn() -> bool,
        ) -> CandleResult<Option<Vec<f32>>> {
            if cancel() {
                return Ok(None);
            }
            if let Some(n) = self.cancel_after {
                if frames.len() > n {
                    return Ok(None);
                }
            }
            let mut pcm = Vec::with_capacity(frames.len() * self.spf);
            for f in frames {
                let v = f[0] as f32;
                pcm.extend(std::iter::repeat_n(v, self.spf));
            }
            Ok(Some(pcm))
        }
    }

    fn frame(cb0: u32) -> RvqFrame {
        vec![cb0, 0, 0, 0]
    }

    /// The one-shot reference: a single full-length decode over every frame.
    fn full_decode(codec: &FakeCodec, frames: &[RvqFrame]) -> Vec<f32> {
        codec.decode_prefix(frames, &|| false).unwrap().unwrap()
    }

    #[test]
    fn interleaved_stream_emits_multiple_chunks_first_well_before_last() {
        // 20 frames, block 8 → boundaries at 8, 16, and a finish flush at 20 → 3 chunks.
        let codec = FakeCodec {
            spf: 10,
            cancel_after: None,
        };
        let frames: Vec<RvqFrame> = (0..20).map(|i| frame(i + 1)).collect();
        let no_cancel = || false;

        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, 8);
        for f in &frames {
            chunker
                .push(f.clone(), &no_cancel, &mut |c| chunks.push(c))
                .unwrap()
                .unwrap();
        }
        let track = chunker
            .finish(&no_cancel, &mut |c| chunks.push(c))
            .unwrap()
            .unwrap();

        // Genuinely incremental: >= 2 chunks, the first emitted after only the first block (so its
        // samples cover far less than the whole track), and none carries the entire track.
        assert!(
            chunks.len() >= 2,
            "expected >= 2 chunks, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].index, 0);
        assert_eq!(
            chunks[0].samples.len(),
            8 * codec.spf,
            "first chunk is exactly the first block of frames"
        );
        assert!(
            chunks.iter().all(|c| c.samples.len() < track.samples.len()),
            "no single chunk holds the whole track"
        );
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i, "contiguous 0..N indices");
            assert_eq!(c.sample_rate, 24_000);
            assert_eq!(c.channels, 1);
        }

        // Reassembly law: concat(chunks) == the returned track == a single full-length decode.
        let reassembled: Vec<f32> = chunks.iter().flat_map(|c| c.samples.clone()).collect();
        assert_eq!(reassembled, track.samples, "reassembly law");
        assert_eq!(
            track.samples,
            full_decode(&codec, &frames),
            "the streamed track is byte-identical to one full-length decode"
        );
    }

    #[test]
    fn remainder_below_a_full_block_is_flushed_at_finish() {
        // 10 frames, block 8 → one boundary at 8, then a finish flush for the last 2 frames.
        let codec = FakeCodec {
            spf: 4,
            cancel_after: None,
        };
        let frames: Vec<RvqFrame> = (0..10).map(|i| frame(i + 1)).collect();
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, 8);
        for f in &frames {
            chunker
                .push(f.clone(), &no_cancel, &mut |c| chunks.push(c))
                .unwrap()
                .unwrap();
        }
        let track = chunker
            .finish(&no_cancel, &mut |c| chunks.push(c))
            .unwrap()
            .unwrap();
        assert_eq!(chunks.len(), 2, "one full block + a remainder chunk");
        assert_eq!(chunks[0].samples.len(), 8 * codec.spf);
        assert_eq!(chunks[1].samples.len(), 2 * codec.spf);
        let reassembled: Vec<f32> = chunks.iter().flat_map(|c| c.samples.clone()).collect();
        assert_eq!(reassembled, track.samples);
    }

    #[test]
    fn exact_multiple_of_block_does_not_double_decode_at_finish() {
        // 16 frames, block 8 → boundaries at 8 and 16; finish must not emit a redundant empty chunk.
        let codec = FakeCodec {
            spf: 2,
            cancel_after: None,
        };
        let frames: Vec<RvqFrame> = (0..16).map(|i| frame(i + 1)).collect();
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, 8);
        for f in &frames {
            chunker
                .push(f.clone(), &no_cancel, &mut |c| chunks.push(c))
                .unwrap();
        }
        let track = chunker
            .finish(&no_cancel, &mut |c| chunks.push(c))
            .unwrap()
            .unwrap();
        assert_eq!(
            chunks.len(),
            2,
            "exactly two full-block chunks, no trailing empty"
        );
        let reassembled: Vec<f32> = chunks.iter().flat_map(|c| c.samples.clone()).collect();
        assert_eq!(reassembled, track.samples);
    }

    #[test]
    fn cancel_inside_a_block_decode_stops_the_stream() {
        // The codec cancels once the prefix grows beyond 8 frames: the first block flush (at 8)
        // succeeds, the second (at 16) returns None → push reports cancellation.
        let codec = FakeCodec {
            spf: 3,
            cancel_after: Some(8),
        };
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, 8);
        let mut canceled = false;
        for i in 0..20u32 {
            if chunker
                .push(frame(i + 1), &no_cancel, &mut |c| chunks.push(c))
                .unwrap()
                .is_none()
            {
                canceled = true;
                break;
            }
        }
        assert!(canceled, "the codec cancel must surface through push()");
        assert_eq!(chunks.len(), 1, "only the first block's chunk was emitted");
    }

    #[test]
    fn empty_stream_yields_no_chunks_and_an_empty_track() {
        let codec = FakeCodec {
            spf: 4,
            cancel_after: None,
        };
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let chunker = StreamingChunker::new(&codec, 8);
        let track = chunker
            .finish(&no_cancel, &mut |c| chunks.push(c))
            .unwrap()
            .unwrap();
        assert!(chunks.is_empty());
        assert!(track.samples.is_empty());
    }
}
