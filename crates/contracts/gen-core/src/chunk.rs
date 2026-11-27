//! Shared bounded chunked output delivery contract (`PIPE-16`, `§6.9`).
//!
//! Provides ONE shared bounded [`ChunkSink`] / `emit` primitive used uniformly by:
//!
//! - single-terminal whole-clip batch delivery ([`TerminalChunkCollector`]), which moves
//!   [`GenerationOutput::Video`](crate::GenerationOutput::Video) directly to the caller as one
//!   terminal chunk (`sequence = 0`, `is_final = true`) with **zero** extra full-clip copies; and
//! - bounded multi-chunk incremental delivery ([`BoundedChunkSink`]), enforcing strict monotone
//!   sequence numbering from `0`, terminal `is_final = true` sealing, cooperative [`CancelFlag`]
//!   cancellation, and explicit backpressure ([`BackpressureError`]) when a lagging consumer fills
//!   `max_buffered_chunks`.
//!
//! Zero tensor dependencies: operates purely over generic host payloads `T`.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::{CancelFlag, Error, Result};

/// Output residency / buffering declaration carried by an [`OutputChunk`] (`PIPE-16`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkResidencyPolicy {
    /// Whole-clip batch delivery: one terminal chunk (`sequence = 0, is_final = true`) handed
    /// directly to the consumer without allocating or retaining another full-clip buffer.
    TerminalWholeClip,
    /// Bounded incremental delivery retaining at most `max_buffered_chunks` chunks in flight before
    /// exerting backpressure on the producer.
    BoundedStream { max_buffered_chunks: usize },
}

impl ChunkResidencyPolicy {
    /// Validate that any bounded capacity is non-zero.
    pub fn validate(&self) -> Result<()> {
        if let Self::BoundedStream {
            max_buffered_chunks: 0,
        } = self
        {
            return Err(Error::Unsupported(
                "chunk residency policy requires max_buffered_chunks >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// Causal codec / KV-cache residency policy associated with an emitted [`OutputChunk`] (`PIPE-16`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalCodecStatePolicy {
    /// No persistent cross-chunk causal KV or codec state is retained past terminal delivery.
    Stateless,
    /// Rolling causal KV / codec window bounded to `max_attention_tokens` plus `sink_tokens`.
    RollingKvWindow {
        max_attention_tokens: usize,
        sink_tokens: usize,
    },
    /// Full-sequence causal KV / codec state retained across all chunks in the clip.
    FullSequenceKv,
}

/// One typed output unit delivered through a [`ChunkSink`] (`PIPE-16`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputChunk<T> {
    /// Strict 0-based monotone sequence ordinal within the request (`0, 1, ..., N-1`).
    pub sequence: u64,
    /// `true` strictly on the final chunk of the request; no chunk may follow a terminal chunk.
    pub is_final: bool,
    /// Owned chunk payload moved through the sink without cloning.
    pub payload: T,
    /// Optional residency/buffering policy declaration for this delivery.
    pub residency_policy: Option<ChunkResidencyPolicy>,
    /// Optional causal KV / codec state residency declaration for this delivery.
    pub causal_codec_policy: Option<CausalCodecStatePolicy>,
}

impl<T> OutputChunk<T> {
    /// Construct an ordered chunk at `sequence` carrying `payload`.
    pub fn new(sequence: u64, is_final: bool, payload: T) -> Self {
        Self {
            sequence,
            is_final,
            payload,
            residency_policy: None,
            causal_codec_policy: None,
        }
    }

    /// Construct the single whole-clip terminal chunk (`sequence = 0`, `is_final = true`) with
    /// [`ChunkResidencyPolicy::TerminalWholeClip`].
    pub fn terminal(payload: T) -> Self {
        Self {
            sequence: 0,
            is_final: true,
            payload,
            residency_policy: Some(ChunkResidencyPolicy::TerminalWholeClip),
            causal_codec_policy: None,
        }
    }

    /// Attach a [`ChunkResidencyPolicy`] declaration to this chunk.
    pub fn with_residency_policy(mut self, policy: ChunkResidencyPolicy) -> Self {
        self.residency_policy = Some(policy);
        self
    }

    /// Attach a [`CausalCodecStatePolicy`] declaration to this chunk.
    pub fn with_causal_codec_policy(mut self, policy: CausalCodecStatePolicy) -> Self {
        self.causal_codec_policy = Some(policy);
        self
    }
}

/// Backpressure policy applied when a [`BoundedChunkSink`] reaches `max_buffered_chunks`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackpressurePolicy {
    /// Reject the incoming emit with [`BackpressureError`] while keeping `next_sequence`
    /// unchanged so the producer can retry once the consumer drains capacity.
    RejectEmit,
}

/// Typed backpressure error returned by [`BoundedChunkSink`] when `buffered >= max_buffered_chunks`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "chunk sink backpressure: buffered={buffered}, max_buffered_chunks={max_buffered_chunks}, rejected sequence={sequence}"
)]
pub struct BackpressureError {
    pub sequence: u64,
    pub buffered: usize,
    pub max_buffered_chunks: usize,
}

impl BackpressureError {
    /// Extract a [`BackpressureError`] from a [`gen_core::Error`](crate::Error), if present.
    pub fn from_error(err: &Error) -> Option<&Self> {
        match err {
            Error::Backend(inner) => inner.downcast_ref::<Self>(),
            _ => None,
        }
    }
}

impl From<BackpressureError> for Error {
    fn from(err: BackpressureError) -> Self {
        Error::backend(err)
    }
}

/// Single shared bounded chunked emission seam (`PIPE-16`, `§6.9`).
pub trait ChunkSink<T> {
    /// Emit one [`OutputChunk<T>`] into this sink.
    ///
    /// Implementations must enforce cooperative cancellation ([`CancelFlag`]), strict monotone
    /// `sequence` starting at `0`, rejection of emits after a terminal `is_final = true` chunk,
    /// and bounded retention / backpressure where applicable.
    fn emit(&mut self, chunk: OutputChunk<T>) -> Result<()>;
}

impl<T, S: ChunkSink<T> + ?Sized> ChunkSink<T> for &mut S {
    fn emit(&mut self, chunk: OutputChunk<T>) -> Result<()> {
        (**self).emit(chunk)
    }
}

impl<T, S: ChunkSink<T> + ?Sized> ChunkSink<T> for Box<S> {
    fn emit(&mut self, chunk: OutputChunk<T>) -> Result<()> {
        (**self).emit(chunk)
    }
}

/// Emit `payload` as a single terminal chunk (`sequence = 0`, `is_final = true`) through `sink`
/// with zero extra copies (`PIPE-16`).
pub fn emit_terminal_chunk<T, S: ChunkSink<T> + ?Sized>(
    sink: &mut S,
    payload: T,
    causal_codec_policy: Option<CausalCodecStatePolicy>,
) -> Result<()> {
    let mut chunk = OutputChunk::terminal(payload);
    chunk.causal_codec_policy = causal_codec_policy;
    sink.emit(chunk)
}

/// Bounded in-memory queue sink enforcing strict monotone sequence order, cancellation, terminal
/// sealing, and explicit backpressure at `max_buffered_chunks` (`PIPE-16`).
#[derive(Debug)]
pub struct BoundedChunkSink<T> {
    queue: VecDeque<OutputChunk<T>>,
    max_buffered_chunks: usize,
    backpressure_policy: BackpressurePolicy,
    cancel: CancelFlag,
    next_sequence: u64,
    finished: bool,
    backpressure_events: u64,
}

impl<T> BoundedChunkSink<T> {
    /// Create a bounded sink with capacity `max_buffered_chunks >= 1` using
    /// [`BackpressurePolicy::RejectEmit`].
    pub fn new(max_buffered_chunks: usize, cancel: CancelFlag) -> Result<Self> {
        Self::with_policy(max_buffered_chunks, BackpressurePolicy::RejectEmit, cancel)
    }

    /// Create a bounded sink with an explicit [`BackpressurePolicy`].
    pub fn with_policy(
        max_buffered_chunks: usize,
        backpressure_policy: BackpressurePolicy,
        cancel: CancelFlag,
    ) -> Result<Self> {
        if max_buffered_chunks == 0 {
            return Err(Error::Unsupported(
                "bounded chunk sink requires max_buffered_chunks >= 1".into(),
            ));
        }
        Ok(Self {
            queue: VecDeque::with_capacity(max_buffered_chunks),
            max_buffered_chunks,
            backpressure_policy,
            cancel,
            next_sequence: 0,
            finished: false,
            backpressure_events: 0,
        })
    }

    /// Configured capacity ceiling in chunks.
    pub fn max_buffered_chunks(&self) -> usize {
        self.max_buffered_chunks
    }

    /// Configured backpressure policy.
    pub fn backpressure_policy(&self) -> BackpressurePolicy {
        self.backpressure_policy
    }

    /// Number of chunks currently buffered awaiting consumer drain.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// `true` when zero chunks are currently buffered.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Next expected monotone 0-based `sequence` ordinal.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// `true` once a terminal chunk (`is_final = true`) has been accepted.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Total number of backpressure rejections observed by this sink.
    pub fn backpressure_events(&self) -> u64 {
        self.backpressure_events
    }

    /// Shared cooperative cancellation flag polled on every `emit`.
    pub fn cancel_flag(&self) -> &CancelFlag {
        &self.cancel
    }

    /// Pop the oldest buffered chunk in strict sequence order.
    pub fn pop(&mut self) -> Option<OutputChunk<T>> {
        self.queue.pop_front()
    }

    /// Drain all currently buffered chunks in strict sequence order.
    pub fn drain(&mut self) -> Vec<OutputChunk<T>> {
        self.queue.drain(..).collect()
    }

    /// Drain at most `max` buffered chunks in strict sequence order.
    pub fn drain_up_to(&mut self, max: usize) -> Vec<OutputChunk<T>> {
        let take = max.min(self.queue.len());
        self.queue.drain(..take).collect()
    }
}

impl<T> ChunkSink<T> for BoundedChunkSink<T> {
    fn emit(&mut self, chunk: OutputChunk<T>) -> Result<()> {
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(policy) = chunk.residency_policy {
            policy.validate()?;
        }
        if self.finished {
            return Err(Error::Unsupported(
                "chunk sink cannot accept chunks after a terminal chunk (is_final = true)".into(),
            ));
        }
        if chunk.sequence != self.next_sequence {
            return Err(Error::Unsupported(format!(
                "chunk sink sequence mismatch: expected {}, got {}",
                self.next_sequence, chunk.sequence
            )));
        }
        if self.queue.len() >= self.max_buffered_chunks {
            self.backpressure_events = self.backpressure_events.saturating_add(1);
            match self.backpressure_policy {
                BackpressurePolicy::RejectEmit => {
                    return Err(BackpressureError {
                        sequence: chunk.sequence,
                        buffered: self.queue.len(),
                        max_buffered_chunks: self.max_buffered_chunks,
                    }
                    .into());
                }
            }
        }
        let next = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| Error::Msg("chunk sink sequence counter overflow".into()))?;
        self.next_sequence = next;
        if chunk.is_final {
            self.finished = true;
        }
        self.queue.push_back(chunk);
        Ok(())
    }
}

/// Zero-copy whole-clip batch collector accepting strictly one terminal chunk
/// (`sequence = 0`, `is_final = true`) and moving `T` directly into the caller (`PIPE-16`).
///
/// Requires no `Clone` bound on `T`: the emitted payload is stored in-place and moved out via
/// [`into_output`](Self::into_output) without cloning or copying buffers.
#[derive(Debug)]
pub struct TerminalChunkCollector<T> {
    cancel: CancelFlag,
    slot: Option<OutputChunk<T>>,
}

impl<T> Default for TerminalChunkCollector<T> {
    fn default() -> Self {
        Self::new(CancelFlag::new())
    }
}

impl<T> TerminalChunkCollector<T> {
    /// Create an empty terminal chunk collector polling `cancel`.
    pub fn new(cancel: CancelFlag) -> Self {
        Self { cancel, slot: None }
    }

    /// `true` once the single terminal chunk has been emitted.
    pub fn is_ready(&self) -> bool {
        self.slot.is_some()
    }

    /// Borrow the accepted terminal chunk without moving it.
    pub fn chunk(&self) -> Option<&OutputChunk<T>> {
        self.slot.as_ref()
    }

    /// Shared cooperative cancellation flag polled on `emit`.
    pub fn cancel_flag(&self) -> &CancelFlag {
        &self.cancel
    }

    /// Move the accepted terminal [`OutputChunk<T>`] out of the collector with zero copying.
    pub fn into_chunk(self) -> Result<OutputChunk<T>> {
        self.slot
            .ok_or_else(|| Error::Msg("terminal chunk collector received no terminal chunk".into()))
    }

    /// Move the payload `T` directly out of the collector with zero cloning/copying.
    pub fn into_output(self) -> Result<T> {
        Ok(self.into_chunk()?.payload)
    }
}

impl<T> ChunkSink<T> for TerminalChunkCollector<T> {
    fn emit(&mut self, chunk: OutputChunk<T>) -> Result<()> {
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(policy) = chunk.residency_policy {
            policy.validate()?;
        }
        if self.slot.is_some() {
            return Err(Error::Unsupported(
                "terminal chunk collector cannot accept another chunk after terminal emission"
                    .into(),
            ));
        }
        if chunk.sequence != 0 {
            return Err(Error::Unsupported(format!(
                "terminal chunk collector requires sequence == 0, got {}",
                chunk.sequence
            )));
        }
        if !chunk.is_final {
            return Err(Error::Unsupported(
                "terminal chunk collector requires is_final == true".into(),
            ));
        }
        self.slot = Some(chunk);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationOutput, Image};

    struct MockChunkProducer {
        total_chunks: u64,
        next_to_emit: u64,
        emitted_sequences: Vec<u64>,
    }

    impl MockChunkProducer {
        fn new(total_chunks: u64) -> Self {
            Self {
                total_chunks,
                next_to_emit: 0,
                emitted_sequences: Vec::new(),
            }
        }

        fn emit_next<S: ChunkSink<Vec<u8>> + ?Sized>(&mut self, sink: &mut S) -> Result<bool> {
            if self.next_to_emit >= self.total_chunks {
                return Ok(false);
            }
            let seq = self.next_to_emit;
            let is_final = seq + 1 == self.total_chunks;
            let chunk = OutputChunk::new(seq, is_final, vec![seq as u8; 16])
                .with_residency_policy(ChunkResidencyPolicy::BoundedStream {
                    max_buffered_chunks: 4,
                })
                .with_causal_codec_policy(CausalCodecStatePolicy::RollingKvWindow {
                    max_attention_tokens: 9360,
                    sink_tokens: 0,
                });
            sink.emit(chunk)?;
            self.emitted_sequences.push(seq);
            self.next_to_emit += 1;
            Ok(true)
        }
    }

    #[test]
    fn mocked_producer_enforces_strict_monotone_ordering_and_terminal_completion() {
        let cancel = CancelFlag::new();
        let mut sink = BoundedChunkSink::new(8, cancel).unwrap();
        let mut producer = MockChunkProducer::new(5);

        while producer.emit_next(&mut sink).unwrap() {}
        assert!(sink.is_finished());
        assert_eq!(sink.next_sequence(), 5);
        assert_eq!(producer.emitted_sequences, vec![0, 1, 2, 3, 4]);

        let drained = sink.drain();
        assert_eq!(drained.len(), 5);
        for (expected, chunk) in drained.iter().enumerate() {
            assert_eq!(chunk.sequence, expected as u64);
            assert_eq!(chunk.is_final, expected == 4);
            assert_eq!(chunk.payload, vec![expected as u8; 16]);
        }

        // Emitting anything after is_final = true must fail loudly.
        let post_final = sink
            .emit(OutputChunk::new(5, true, vec![99]))
            .expect_err("emit after terminal chunk must fail");
        assert!(matches!(post_final, Error::Unsupported(_)));

        // Fresh sink rejects non-zero start, gaps, and duplicates.
        let mut strict = BoundedChunkSink::new(4, CancelFlag::new()).unwrap();
        assert!(matches!(
            strict.emit(OutputChunk::new(1, false, vec![1])),
            Err(Error::Unsupported(_))
        ));
        strict.emit(OutputChunk::new(0, false, vec![0])).unwrap();
        assert!(matches!(
            strict.emit(OutputChunk::new(0, false, vec![0])),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            strict.emit(OutputChunk::new(2, true, vec![2])),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            BoundedChunkSink::<Vec<u8>>::new(0, CancelFlag::new()),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn bounded_sink_exerts_backpressure_when_consumer_lags_and_resumes_after_drain() {
        let cancel = CancelFlag::new();
        let mut sink = BoundedChunkSink::new(2, cancel).unwrap();
        let mut producer = MockChunkProducer::new(5);

        // Producer emits 2 chunks filling the capacity=2 ring/queue.
        assert!(producer.emit_next(&mut sink).unwrap());
        assert!(producer.emit_next(&mut sink).unwrap());
        assert_eq!(sink.len(), 2);
        assert_eq!(sink.next_sequence(), 2);

        // Consumer lags: the 3rd emit (sequence=2) hits backpressure without advancing sequence or
        // growing the buffer beyond max_buffered_chunks.
        let err = producer
            .emit_next(&mut sink)
            .expect_err("full bounded sink must exert backpressure");
        let bp = BackpressureError::from_error(&err).expect("typed BackpressureError");
        assert_eq!(
            *bp,
            BackpressureError {
                sequence: 2,
                buffered: 2,
                max_buffered_chunks: 2,
            }
        );
        assert_eq!(sink.len(), 2);
        assert_eq!(sink.next_sequence(), 2);
        assert_eq!(sink.backpressure_events(), 1);
        assert!(!sink.is_finished());

        // Consumer drains 1 chunk; producer retries sequence=2 and succeeds.
        let first = sink.pop().unwrap();
        assert_eq!(first.sequence, 0);
        assert!(!first.is_final);
        assert!(producer.emit_next(&mut sink).unwrap());
        assert_eq!(sink.len(), 2);

        // Drain the rest in lockstep until terminal chunk 4 completes.
        let mut received = vec![first];
        received.extend(sink.drain());
        while producer.emit_next(&mut sink).unwrap() {
            received.extend(sink.drain_up_to(1));
        }
        assert!(sink.is_finished());
        assert!(sink.is_empty());
        assert_eq!(
            received.iter().map(|c| c.sequence).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert!(received[..4].iter().all(|c| !c.is_final));
        assert!(received[4].is_final);
    }

    #[test]
    fn sink_cancellation_mid_stream_stops_further_emits_cleanly() {
        // Pre-cancelled bounded sink rejects chunk 0 with Error::Canceled and buffers nothing.
        let pre_cancel = CancelFlag::new();
        pre_cancel.cancel();
        let mut pre_sink = BoundedChunkSink::new(2, pre_cancel).unwrap();
        let mut pre_producer = MockChunkProducer::new(3);
        assert!(matches!(
            pre_producer.emit_next(&mut pre_sink),
            Err(Error::Canceled)
        ));
        assert_eq!(pre_sink.len(), 0);
        assert_eq!(pre_sink.next_sequence(), 0);

        // Mid-stream cancel stops subsequent emits cleanly without advancing past the cut.
        let cancel = CancelFlag::new();
        let mut sink = BoundedChunkSink::new(4, cancel.clone()).unwrap();
        let mut producer = MockChunkProducer::new(5);
        assert!(producer.emit_next(&mut sink).unwrap());
        assert!(producer.emit_next(&mut sink).unwrap());
        cancel.cancel();
        let err = producer
            .emit_next(&mut sink)
            .expect_err("cancelled sink must reject mid-stream emit");
        assert!(matches!(err, Error::Canceled));
        assert_eq!(sink.len(), 2);
        assert!(!sink.is_finished());
        assert_eq!(
            sink.drain()
                .into_iter()
                .map(|c| c.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        // Terminal collector also honours cancellation before accepting.
        let term_cancel = CancelFlag::new();
        term_cancel.cancel();
        let mut collector = TerminalChunkCollector::new(term_cancel);
        assert!(matches!(
            emit_terminal_chunk(&mut collector, vec![1u8, 2, 3], None),
            Err(Error::Canceled)
        ));
        assert!(!collector.is_ready());
    }

    #[test]
    fn terminal_collector_moves_non_clone_and_video_frames_with_zero_copy() {
        // Non-Clone payload proves move-only semantics at compile time and buffer pointer identity.
        #[derive(Debug, PartialEq, Eq)]
        struct NonCloneClip {
            bytes: Vec<u8>,
            tag: u64,
        }

        let buf = vec![10u8, 20, 30, 40, 50];
        let expected_buf_ptr = buf.as_ptr();
        let clip = NonCloneClip {
            bytes: buf,
            tag: 0xDEAD_BEEF,
        };
        let mut collector = TerminalChunkCollector::new(CancelFlag::new());
        emit_terminal_chunk(
            &mut collector,
            clip,
            Some(CausalCodecStatePolicy::RollingKvWindow {
                max_attention_tokens: 9360,
                sink_tokens: 0,
            }),
        )
        .unwrap();
        let chunk = collector.chunk().unwrap();
        assert_eq!(chunk.sequence, 0);
        assert!(chunk.is_final);
        assert_eq!(
            chunk.residency_policy,
            Some(ChunkResidencyPolicy::TerminalWholeClip)
        );
        assert_eq!(
            chunk.causal_codec_policy,
            Some(CausalCodecStatePolicy::RollingKvWindow {
                max_attention_tokens: 9360,
                sink_tokens: 0,
            })
        );
        let moved = collector.into_output().unwrap();
        assert_eq!(
            moved.bytes.as_ptr(),
            expected_buf_ptr,
            "non-Clone payload Vec buffer must move with zero copy"
        );
        assert_eq!(moved.tag, 0xDEAD_BEEF);

        // GenerationOutput::Video zero-copy pointer identity on both outer Vec<Image> and pixel Vec<u8>.
        let frame_pixels = vec![42u8; 16 * 16 * 3];
        let expected_pixel_ptr = frame_pixels.as_ptr();
        let frames = vec![Image {
            width: 16,
            height: 16,
            pixels: frame_pixels,
        }];
        let expected_frames_ptr = frames.as_ptr();

        let mut video_collector = TerminalChunkCollector::new(CancelFlag::new());
        emit_terminal_chunk(
            &mut video_collector,
            GenerationOutput::Video {
                frames,
                fps: 16,
                audio: None,
            },
            Some(CausalCodecStatePolicy::RollingKvWindow {
                max_attention_tokens: 9360,
                sink_tokens: 0,
            }),
        )
        .unwrap();
        let out = video_collector.into_output().unwrap();
        match out {
            GenerationOutput::Video { frames, fps, audio } => {
                assert_eq!(fps, 16);
                assert!(audio.is_none());
                assert_eq!(
                    frames.as_ptr(),
                    expected_frames_ptr,
                    "whole-clip frames Vec must move without reallocation or copy"
                );
                assert_eq!(
                    frames[0].pixels.as_ptr(),
                    expected_pixel_ptr,
                    "frame pixel buffer must move without copy"
                );
            }
            other => panic!("expected Video output, got {other:?}"),
        }
    }

    #[test]
    fn terminal_collector_rejects_non_zero_non_final_and_duplicate_chunks() {
        let mut collector = TerminalChunkCollector::new(CancelFlag::new());
        assert!(matches!(
            collector.emit(OutputChunk::new(1, true, 7u32)),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            collector.emit(OutputChunk::new(0, false, 7u32)),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            collector.emit(OutputChunk::terminal(7u32).with_residency_policy(
                ChunkResidencyPolicy::BoundedStream {
                    max_buffered_chunks: 0
                }
            )),
            Err(Error::Unsupported(_))
        ));
        collector.emit(OutputChunk::terminal(7u32)).unwrap();
        assert!(matches!(
            collector.emit(OutputChunk::terminal(8u32)),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(collector.into_output().unwrap(), 7);
    }
}
