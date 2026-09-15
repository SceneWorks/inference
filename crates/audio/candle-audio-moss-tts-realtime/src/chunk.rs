//! Interleaved streaming: RVQ frames → PCM chunks, decoded block-wise **as the AR loop runs**
//! (sc-13392).
//!
//! MOSS-TTS-Realtime is autoregressive: [`crate::decode`]'s AR loop emits one RVQ frame at a time.
//! Rather than wait for the whole track and chunk it after the fact, [`StreamingChunker`] retains one
//! request-local codec state and hands the codec only the newly available frames on each
//! scheduled boundary. The codec emits only that block's new PCM, so chunks flow **while** the AR loop
//! is still generating later frames without replaying the growing prefix. [`crate::model`] feeds it
//! each frame from inside [`crate::decode::Decoder::run`].
//!
//! The returned track is exactly the **concatenation of the emitted blocks**. The `AudioChunk`
//! reassembly law (concat(chunks) == track) therefore holds by construction, and one-shot
//! `generate()` is byte-identical to the concatenated stream because both entry points use this one
//! stateful block sequence for the same request+seed.
//!
//! The codec is abstracted behind [`IncrementalDecoder`] so this interleaving logic and its linear
//! work bound are unit-testable **offline** (no weights, no real codec) with a fake frame→PCM source.

use candle_audio::candle_core::Result as CandleResult;
use candle_audio::gen_core::{AudioChunk, AudioTrack};

use crate::codec::DecodePartitionSchedule;
use crate::decode::RvqFrame;

/// Decodes newly appended RVQ frames with request-local state. The production implementation is the
/// MOSS-Audio-Tokenizer codec ([`crate::codec::MossAudioCodec`]); a fake drives the offline
/// [`StreamingChunker`] tests.
pub trait IncrementalDecoder {
    /// State retained for exactly one synthesis request. Implementations keep it bounded by their
    /// fixed causal context rather than the number of frames already decoded.
    type State;

    /// Native PCM sample rate (Hz).
    fn sample_rate(&self) -> u32;
    /// Waveform samples produced per RVQ frame (the codec's downsample rate).
    fn samples_per_frame(&self) -> usize;

    /// Create a fresh request state. Sharing one loaded decoder across requests must not share their
    /// mutable causal history.
    fn new_state(&self) -> Self::State;

    /// Decode only the newly appended `frames`, returning exactly
    /// `frames.len() * samples_per_frame()` new mono samples. `state` carries the bounded causal
    /// history and absolute positions from prior calls. Returns `Ok(None)` if `cancel` tripped.
    fn decode_next(
        &self,
        state: &mut Self::State,
        frames: &[RvqFrame],
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Vec<f32>>>;
}

/// Block-wise interleaved chunker (see the module docs). Feed it AR frames with [`push`](Self::push)
/// as they are produced; call [`finish`](Self::finish) at EOS/budget to flush the remainder and get
/// the full [`AudioTrack`]. It emits `0..N`-indexed, contiguous [`AudioChunk`]s whose concatenation
/// is exactly the track (the reassembly law).
pub struct StreamingChunker<'a, D: IncrementalDecoder> {
    decoder: &'a D,
    /// Request-local codec history (stage positions + bounded per-layer KV state in production).
    state: D::State,
    schedule: DecodePartitionSchedule,
    /// Only frames not yet handed to the codec. Cleared after every successful flush.
    frames: Vec<RvqFrame>,
    /// The concatenation of every emitted block — the running track PCM (the reassembly law).
    samples: Vec<f32>,
    /// Next chunk index (`0..N`).
    index: usize,
}

impl<'a, D: IncrementalDecoder> StreamingChunker<'a, D> {
    /// New chunker using the request's canonical, validated codec partition schedule.
    pub fn new(decoder: &'a D, schedule: DecodePartitionSchedule) -> Self {
        Self {
            decoder,
            state: decoder.new_state(),
            schedule,
            frames: Vec::new(),
            samples: Vec::new(),
            index: 0,
        }
    }

    /// Buffer one AR frame; on the next scheduled boundary decode only that new block and emit its PCM
    /// as the next chunk. Returns `Ok(None)` if `cancel` tripped inside the codec decode.
    pub fn push(
        &mut self,
        frame: RvqFrame,
        cancel: &dyn Fn() -> bool,
        on_chunk: &mut dyn FnMut(AudioChunk),
    ) -> CandleResult<Option<()>> {
        self.frames.push(frame);
        if self.frames.len() == self.schedule.block_frames(self.index)
            && self.flush(cancel, on_chunk)?.is_none()
        {
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
        if !self.frames.is_empty() && self.flush(cancel, on_chunk)?.is_none() {
            return Ok(None);
        }
        Ok(Some(AudioTrack {
            samples: self.samples,
            sample_rate: self.decoder.sample_rate(),
            channels: 1,
            stems: Vec::new(),
        }))
    }

    /// Decode the current pending block with the request-local state and emit its new PCM as one
    /// chunk. `Ok(None)` on cancel.
    fn flush(
        &mut self,
        cancel: &dyn Fn() -> bool,
        on_chunk: &mut dyn FnMut(AudioChunk),
    ) -> CandleResult<Option<()>> {
        let pcm = match self
            .decoder
            .decode_next(&mut self.state, &self.frames, cancel)?
        {
            Some(p) => p,
            None => return Ok(None),
        };
        let expected = self
            .frames
            .len()
            .checked_mul(self.decoder.samples_per_frame())
            .ok_or_else(|| {
                candle_audio::candle_core::Error::Msg(
                    "incremental codec output length overflow".to_owned(),
                )
            })?;
        if pcm.len() != expected {
            candle_audio::candle_core::bail!(
                "incremental codec returned {} samples for {} frames; expected {expected}",
                pcm.len(),
                self.frames.len()
            );
        }
        self.frames.clear();
        if !pcm.is_empty() {
            on_chunk(AudioChunk {
                samples: pcm.clone(),
                sample_rate: self.decoder.sample_rate(),
                channels: 1,
                index: self.index,
            });
            self.samples.extend_from_slice(&pcm);
            self.index += 1;
        }
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn uniform_schedule(block: usize) -> DecodePartitionSchedule {
        DecodePartitionSchedule::new(block, block).unwrap()
    }

    /// A fake stateful frame→PCM codec: frame `f`'s codebook-0 code `c` contributes `spf` samples all
    /// equal to `c as f32`. `work` counts every frame actually handed to the codec, so replaying old
    /// prefixes makes the deterministic linear-work assertion fail.
    struct FakeCodec {
        spf: usize,
        cancel_after: Option<usize>,
        work: Cell<usize>,
    }

    #[derive(Default)]
    struct FakeState {
        decoded_frames: usize,
    }

    impl IncrementalDecoder for FakeCodec {
        type State = FakeState;

        fn sample_rate(&self) -> u32 {
            24_000
        }
        fn samples_per_frame(&self) -> usize {
            self.spf
        }

        fn new_state(&self) -> Self::State {
            FakeState::default()
        }

        fn decode_next(
            &self,
            state: &mut Self::State,
            frames: &[RvqFrame],
            cancel: &dyn Fn() -> bool,
        ) -> CandleResult<Option<Vec<f32>>> {
            if cancel() {
                return Ok(None);
            }
            if let Some(n) = self.cancel_after {
                if state.decoded_frames + frames.len() > n {
                    return Ok(None);
                }
            }
            self.work.set(self.work.get() + frames.len());
            let mut pcm = Vec::with_capacity(frames.len() * self.spf);
            for f in frames {
                let v = f[0] as f32;
                pcm.extend(std::iter::repeat_n(v, self.spf));
            }
            state.decoded_frames += frames.len();
            Ok(Some(pcm))
        }
    }

    fn frame(cb0: u32) -> RvqFrame {
        vec![cb0, 0, 0, 0]
    }

    /// The one-shot reference: a single full-length decode over every frame.
    fn full_decode(codec: &FakeCodec, frames: &[RvqFrame]) -> Vec<f32> {
        frames
            .iter()
            .flat_map(|frame| std::iter::repeat_n(frame[0] as f32, codec.spf))
            .collect()
    }

    #[test]
    fn interleaved_stream_emits_multiple_chunks_first_well_before_last() {
        // 20 frames, block 8 → boundaries at 8, 16, and a finish flush at 20 → 3 chunks.
        let codec = FakeCodec {
            spf: 10,
            cancel_after: None,
            work: Cell::new(0),
        };
        let frames: Vec<RvqFrame> = (0..20).map(|i| frame(i + 1)).collect();
        let no_cancel = || false;

        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, uniform_schedule(8));
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
            "the concatenated incremental blocks are byte-identical to one-shot chunked decode"
        );
        assert_eq!(
            codec.work.get(),
            frames.len(),
            "each frame is decoded exactly once; growing-prefix replay would be superlinear"
        );
    }

    #[test]
    fn remainder_below_a_full_block_is_flushed_at_finish() {
        // 10 frames, block 8 → one boundary at 8, then a finish flush for the last 2 frames.
        let codec = FakeCodec {
            spf: 4,
            cancel_after: None,
            work: Cell::new(0),
        };
        let frames: Vec<RvqFrame> = (0..10).map(|i| frame(i + 1)).collect();
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, uniform_schedule(8));
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
            work: Cell::new(0),
        };
        let frames: Vec<RvqFrame> = (0..16).map(|i| frame(i + 1)).collect();
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, uniform_schedule(8));
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
        // The codec cancels once total decoded work would exceed 8 frames: the first block flush
        // succeeds, the second returns None → push reports cancellation.
        let codec = FakeCodec {
            spf: 3,
            cancel_after: Some(8),
            work: Cell::new(0),
        };
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let mut chunker = StreamingChunker::new(&codec, uniform_schedule(8));
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
            work: Cell::new(0),
        };
        let no_cancel = || false;
        let mut chunks: Vec<AudioChunk> = Vec::new();
        let chunker = StreamingChunker::new(&codec, uniform_schedule(8));
        let track = chunker
            .finish(&no_cancel, &mut |c| chunks.push(c))
            .unwrap()
            .unwrap();
        assert!(chunks.is_empty());
        assert!(track.samples.is_empty());
    }
}
